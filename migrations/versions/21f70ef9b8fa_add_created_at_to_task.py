"""add created_at to Task

Revision ID: 21f70ef9b8fa
Revises: b412ad3b2060
Create Date: 2025-07-23 16:36:25.867528

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '21f70ef9b8fa'
down_revision = 'b412ad3b2060'
branch_labels = None
depends_on = None


def upgrade():
    # Add the column with a default for existing rows
    with op.batch_alter_table('task', schema=None) as batch_op:
        batch_op.add_column(sa.Column('created_at', sa.DateTime(), nullable=False, server_default='2025-07-23 00:00:00'))
    # Remove the server_default so future inserts use the model default
    with op.batch_alter_table('task', schema=None) as batch_op:
        batch_op.alter_column('created_at', server_default=None)

    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table('task', schema=None) as batch_op:
        batch_op.drop_column('created_at')

    # ### end Alembic commands ###
